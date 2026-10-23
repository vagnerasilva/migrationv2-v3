# Relatório: outras migrações do SageMaker Python SDK v2 para v3

**Público:** cientistas de dados com projetos existentes em v2 que precisam
planejar uma migração para v3.

**Escopo:** este documento complementa os relatórios de Training Jobs e
Processing Jobs. Aqui o foco são os demais componentes: inferência,
deploy, pipelines, tuning, batch transform, registry, Feature Store,
monitoramento, recursos de baixo nível e novas capacidades.

**Versão observada:** `VERSION` neste checkout informa `3.27.0`. Os exemplos,
imports e APIs descritos devem ser conferidos contra a versão que a equipe
pretende instalar e contra os requisitos do serviço AWS na região de destino.

## Índice

1. [Resumo executivo](#1-resumo-executivo)
2. [Como interpretar a migração](#2-como-interpretar-a-migração)
3. [Mudança arquitetural e pacotes](#3-mudança-arquitetural-e-pacotes)
4. [Deploy e inferência online](#4-deploy-e-inferência-online)
5. [Predição, endpoints e serving customizado](#5-predição-endpoints-e-serving-customizado)
6. [Batch Transform](#6-batch-transform)
7. [Pipelines e orquestração](#7-pipelines-e-orquestração)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Model Registry, experimentos e lineage](#9-model-registry-experimentos-e-lineage)
10. [Feature Store](#10-feature-store)
11. [Monitoramento, Clarify e qualidade](#11-monitoramento-clarify-e-qualidade)
12. [SageMaker Core e chamadas de baixo nível](#12-sagemaker-core-e-chamadas-de-baixo-nível)
13. [Novas capacidades e itens sem equivalência direta](#13-novas-capacidades-e-itens-sem-equivalência-direta)
14. [Estratégia de migração para um projeto existente](#14-estratégia-de-migração-para-um-projeto-existente)
15. [Matriz de migração rápida](#15-matriz-de-migração-rápida)
16. [Validação, riscos e checklist](#16-validação-riscos-e-checklist)
17. [Referências do repositório](#17-referências-do-repositório)

## 1. Resumo executivo

O SDK v3 é uma mudança incompatível de interface, não apenas uma atualização
de dependência. As classes v2 `Estimator`, `Model`, `Predictor` e subclasses
não são suportadas como antes. O SDK passa a ser modular e usa interfaces
unificadas, recursos tipados e objetos de configuração.

Para além de Training e Processing:

- `ModelBuilder` unifica a preparação e o deploy de modelos, substituindo a
  composição habitual de `Model`/`Predictor` e modelos específicos de
  framework.
- Pipelines, steps, parâmetros e recursos de workflow passam a ser importados
  principalmente de `sagemaker.mlops` e `sagemaker.core`.
- Hyperparameter tuning, Batch Transform, Model Registry, experimentos,
  Feature Store, Clarify e monitoramento usam APIs e módulos reposicionados.
- V3 adiciona fluxos como `InferenceSpec`, inferência in-process,
  customização de foundation models e componentes novos de Pipeline. Eles não
  devem ser tratados como substitutos automáticos de cada recurso v2.
- O serviço AWS continua sendo o mesmo: IAM, S3, ECR, CloudWatch, capacidade,
  quotas, formatos dos containers e cobrança continuam fazendo parte da
  arquitetura.

Não é necessário migrar todos os componentes de uma aplicação de uma vez.
Crie um ambiente v3 separado, migre um fluxo vertical por vez e valide
resultados, segurança, custos e operação antes de mover workloads de
produção.

## 2. Como interpretar a migração

Este documento usa três tipos de mapeamento:

- **Substituição:** há uma interface v3 que cumpre papel semelhante, mas
  nomes, argumentos, configuração ou comportamento podem diferir.
- **Reposicionamento:** o conceito persiste, mas o módulo ou pacote mudou.
- **Nova capacidade:** não é uma migração 1:1; avalie adoção como feature nova.

Uma linha de equivalência não garante que todos os argumentos, callbacks,
serializadores, opções do serviço ou semânticas da v2 existam na v3. Compare a
assinatura v3 para cada recurso utilizado e teste as rotas que realmente
importam para o projeto.

Os relatórios complementares tratam em detalhe dos fluxos de treinamento e
pré/pós-processamento:

- [Relatório de Training Jobs](./RELATORIO_TRAINING_SDK_V3.md)
- [Relatório de Processing Jobs](./RELATORIO_PROCESSING_SDK_V3.md)

## 3. Mudança arquitetural e pacotes

Na v2, o pacote `sagemaker` concentrava várias APIs. Na v3, as capacidades
estão separadas por responsabilidade:

| Pacote/módulo v3 | Responsabilidade principal |
|---|---|
| `sagemaker-core` (`sagemaker.core`) | Recursos AWS, shapes, sessão, URIs de imagens, parâmetros, transform, Clarify, lineage e utilitários |
| `sagemaker-train` (`sagemaker.train`) | `ModelTrainer`, fine-tuning, tuning e distribuição |
| `sagemaker-serve` (`sagemaker.serve`) | `ModelBuilder`, especificação de inferência, build, registro e deploy |
| `sagemaker-mlops` (`sagemaker.mlops`) | Pipelines, steps, Feature Store e orquestração |

Instalar `sagemaker` instala as dependências do SDK agregado neste checkout;
subpacotes também possuem manifests próprios. Em ambientes controlados, fixe
as versões de modo consistente e evite instalar versões incompatíveis dos
subpacotes diretamente.

Mudanças transversais:

- Prefira imports explícitos por subpacote; o bare `import sagemaker` não
  substitui todos os imports específicos usados em v2.
- `Session` e funções auxiliares como `get_execution_role` passaram para
  `sagemaker.core.helper.session_helper`.
- `image_uris` passou para `sagemaker.core.image_uris`.
- Muitos parâmetros que antes eram argumentos soltos agora pertencem a
  configurações tipadas ou shapes da API.
- Em Pipeline, os valores podem ser expressões/variáveis resolvidas na
  execução. Não os converta ou avalie prematuramente como strings locais.

Exemplo de setup comum:

```python
from sagemaker.core.helper.session_helper import Session, get_execution_role

session = Session()
role = get_execution_role()
region = session.boto_region_name
```

## 4. Deploy e inferência online

### Da composição `Model` + `Predictor` para `ModelBuilder`

O fluxo v2 normalmente construía uma classe de modelo específica (ou `Model`),
chamava `deploy()` e trabalhava com um `Predictor`. A v3 centraliza a
construção no `ModelBuilder`:

| V2 | V3 |
|---|---|
| `sagemaker.model.Model` e subclasses como `PyTorchModel`/`TensorFlowModel` | `sagemaker.serve.ModelBuilder` com imagem, artefato e configuração/spec de inferência |
| `model.deploy(...)` | `model_builder.build(...)` e `model_builder.deploy(...)` |
| `Predictor.predict(...)` | Endpoint retornado pelo fluxo de serving, normalmente `endpoint.invoke(...)` |
| `sagemaker.predictor.Predictor` | recursos/endpoint de `sagemaker.core` e interfaces de serving v3 |

Exemplo conceitual com artefato já treinado:

```python
from sagemaker.serve import ModelBuilder

builder = ModelBuilder(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/inference:tag",
    s3_model_data_url="s3://meu-bucket/modelos/model.tar.gz",
    role_arn=role,
)

model = builder.build(model_name="classificador-v3")
endpoint = builder.deploy(
    endpoint_name="classificador-v3-endpoint",
    instance_type="ml.m5.large",
    initial_instance_count=1,
)
```

Os nomes de argumentos aceitos dependem do caminho de construção usado. Para
um modelo customizado, a API também pode receber `InferenceSpec`,
`SchemaBuilder`, `model_server` ou `model_path`; não combine snippets de
caminhos diferentes sem conferir o construtor correspondente.

### O que validar antes de criar endpoint

- **Artefato:** formato do pacote, diretório raiz, arquivos de modelo e URI
  S3. Confirme que o serving container consegue carregar exatamente esse
  artefato.
- **Imagem de inferência:** versão do framework, arquitetura e aceleradores
  compatíveis com a instância de endpoint.
- **Contrato de inferência:** conteúdo/tipo de entrada, serialização,
  desserialização, formato da resposta e comportamento para inputs inválidos.
- **IAM e rede:** role de execução, acesso a S3/ECR/KMS, VPC, isolamento e
  eventuais chamadas a serviços externos.
- **Capacidade e custo:** tipo/número de instâncias, escalabilidade,
  disponibilidade e limites para endpoint.
- **Ciclo de vida:** endpoints são recursos persistentes e podem continuar
  cobrando depois que a validação terminou. Planeje exclusão ou retenção
  deliberada.

O método `build()` prepara/valida recursos do modelo, enquanto `deploy()`
provisiona a infraestrutura de inferência. Não trate a criação de um modelo
SageMaker como se já fosse um endpoint executável.

## 5. Predição, endpoints e serving customizado

### Inferência customizada

Quando os handlers padrão não representam o modelo, implemente uma
`InferenceSpec` com `load()` e `invoke()` e forneça amostras com
`SchemaBuilder`, conforme o modelo de serving escolhido. Isso substitui parte
dos padrões v2 baseados em `model_fn`, `input_fn`, `predict_fn` e `output_fn`,
mas não significa que o código antigo possa ser migrado sem ajustes.

Revise:

- carregamento do modelo e uso de CPU/GPU;
- parsing do body e compatibilidade dos content types;
- serialização de valores NumPy, tensores, datas ou estruturas aninhadas;
- inicialização lazy versus carga do modelo no startup;
- limites de tamanho, timeout, concorrência e memória;
- dependências incluídas na imagem e no artefato.

### Tipos de endpoint e integração

A v3 documenta workflows com ModelBuilder para endpoints real-time, async,
serverless e multi-model, além de otimização e alguns cenários JumpStart.
Nomes e configuração precisam ser validados no API v3 do cenário escolhido.
Não assuma que cada argumento de `AsyncPredictor`,
`ServerlessInferenceConfig` ou `MultiDataModel` aparece com o mesmo formato
em `ModelBuilder`.

Para modelos treinados na própria sessão, a v3 pode encadear o objeto do
trainer ao `ModelBuilder`; valide que o treinamento terminou e que o artefato
está disponível antes do deploy. A passagem de um trainer diretamente
simplifica o fluxo, mas não elimina a escolha da imagem, do servidor,
dependências e contrato de inferência.

### Endpoints existentes

Para migrar um endpoint ativo, a estratégia segura é criar uma implantação
paralela com nome/recursos controlados, testar tráfego e respostas e só então
alterar roteamento ou clientes. Planeje rollback, compatibilidade da API e
limpeza do endpoint antigo.

## 6. Batch Transform

Batch Transform continua indicado para inferência em lote sobre objetos/datasets
em S3 sem manter um endpoint online. O processor (`Transformer`) está em
`sagemaker.core.transformer`; o modelo que será executado pode ser criado com
`ModelBuilder` ou recursos Core.

| V2 | V3 |
|---|---|
| `sagemaker.transformer.Transformer` | `sagemaker.core.transformer.Transformer` |
| `model.transformer(...)`/`Transformer(...)` | Construção explícita do `Transformer` com `model_name` e configuração de input/output |
| `transformer.transform(...)` | Mantém conceito e nome em v3, com shapes/parâmetros v3 |

Exemplo do padrão documentado:

```python
from sagemaker.core.transformer import Transformer

transformer = Transformer(
    model_name="classificador-v3",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    accept="text/csv",
    assemble_with="Line",
    output_path="s3://meu-bucket/predicoes/",
)

transformer.transform(
    "s3://meu-bucket/entradas/",
    content_type="text/csv",
    split_type="Line",
)
```

Antes de migrar, compare `input_filter`, `join_source`, `output_filter`,
`split_type`, `assemble_with`, compressão, retries, estratégia de batch,
concurrency e comportamento de nomes dos arquivos. Confirme outputs no S3 e
estado do job; batch transform também cria recursos cobrados durante a
execução.

## 7. Pipelines e orquestração

### Imports e composição dos steps

Na v3, `Pipeline` e `ProcessingStep`, `TrainingStep`, `ModelStep`, tuning e
outros steps ficam em `sagemaker.mlops`; contexto de pipeline, parâmetros e
funções ficam em `sagemaker.core.workflow`. O `PipelineSession` é passado aos
produtores de jobs para capturar as requisições sem executar os jobs durante
a definição do pipeline.

| V2 | V3 |
|---|---|
| `sagemaker.workflow.pipeline.Pipeline` | `sagemaker.mlops.workflow.pipeline.Pipeline` |
| `sagemaker.workflow.steps.ProcessingStep` | `sagemaker.mlops.workflow.steps.ProcessingStep` |
| `sagemaker.workflow.steps.TrainingStep` | `sagemaker.mlops.workflow.steps.TrainingStep` |
| `sagemaker.workflow.model_step.ModelStep` | `sagemaker.mlops.workflow.model_step.ModelStep` |
| `sagemaker.workflow.pipeline_context.PipelineSession` | `sagemaker.core.workflow.pipeline_context.PipelineSession` |
| `sagemaker.workflow.parameters.ParameterString` | `sagemaker.core.workflow.parameters.ParameterString` |

Exemplo de construção com um passo de processamento:

```python
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep

pipeline_session = PipelineSession()
# O processor deve receber pipeline_session como sagemaker_session.
step_process = ProcessingStep(
    name="Preprocess",
    step_args=processor.run(code="code/preprocess.py", inputs=..., outputs=...),
)

pipeline = Pipeline(
    name="my-preprocess-pipeline",
    steps=[step_process],
    sagemaker_session=pipeline_session,
)
pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

Use o `step_args` do produtor de job em vez de duplicar manualmente a
requisição. Isso ajuda a manter parâmetros/expressões ligados ao grafo do
pipeline. Para uma migração real, ajuste os `...` com shapes e argumentos
suportados pelos jobs e steps usados pelo projeto.

### O que muda no modo de pensar

- Em notebook interativo, o job é submetido agora; em Pipeline, os mesmos
  métodos constroem uma especificação para execução posterior.
- Parâmetros e propriedades dos steps podem ser valores simbólicos. Não use
  lógica Python local que espere o valor final durante a definição do grafo.
- Revise `depends_on`, condições, paralelismo, cache, retries, parâmetros e
  outputs referenciados por propriedades.
- `upsert()` cria/atualiza a definição e `start()` inicia uma execução:
  mantenha essas ações explícitas em scripts de implantação.
- A v3.27.0 inclui steps mais recentes, como `ForEachStep` e `CleanupStep`;
  capacidades novas precisam de avaliação própria e não são requisito para
  converter pipelines existentes.

Faça uma validação de pipeline em ambiente de teste: compare a definição
serializada, parâmetros, dependências, políticas de execução, outputs e
permissões antes de iniciar workloads de produção.

## 8. Hyperparameter Tuning

Na v2, `HyperparameterTuner` era configurado em torno de um estimator. Na v3,
o tuner usa `ModelTrainer` e as classes de intervalo de parâmetro estão em
`sagemaker.core.parameter`.

| V2 | V3 |
|---|---|
| `sagemaker.tuner.HyperparameterTuner(estimator=...)` | `sagemaker.train.tuner.HyperparameterTuner(model_trainer=...)` |
| `ContinuousParameter`, `IntegerParameter`, `CategoricalParameter` de módulos legados | classes de `sagemaker.core.parameter` |
| Tuning standalone/pipeline | tuner v3 e steps/pipeline de `sagemaker.mlops` |

Exemplo conciso:

```python
from sagemaker.core.parameter import CategoricalParameter, ContinuousParameter
from sagemaker.train.tuner import HyperparameterTuner

tuner = HyperparameterTuner(
    model_trainer=trainer,
    objective_metric_name="validation:loss",
    hyperparameter_ranges={
        "learning_rate": ContinuousParameter(0.0001, 0.01),
        "batch_size": CategoricalParameter([32, 64]),
    },
    metric_definitions=[
        {
            "Name": "validation:loss",
            "Regex": r"validation:loss=([0-9\.]+)",
        }
    ],
    max_jobs=10,
    max_parallel_jobs=2,
    strategy="Bayesian",
    objective_type="Minimize",
)

tuner.tune(inputs=[...], wait=False)
```

Mantenha as métricas consistentes entre o regex e as linhas emitidas pelo
script, escolha objetivo/ranges válidos para o algoritmo e calcule o custo
potencial (`max_jobs`, paralelismo, instâncias e duração). Confirme suporte de
cada parâmetro e de configurações avançadas, como checkpoint/retry, na versão
fixada.

## 9. Model Registry, experimentos e lineage

### Model Registry

`ModelBuilder` pode preparar modelos e registrá-los; recursos de pacotes e
grupos de pacotes também estão disponíveis via `sagemaker.core.resources`.
Migre as etapas de criação de Model Package, aprovação e recuperação de
versões verificando explicitamente metadados, containers, métricas,
content/response types, approval status e permissões.

O padrão geral documentado:

1. Construir o modelo com `ModelBuilder`.
2. Chamar `register()` com o grupo de pacotes e tipos aceitos.
3. Recuperar `ModelPackage` e inspecionar a especificação antes do deploy.

Não assuma que o registro criado pelo método v2 `register()`/`register_model()`
tem a mesma representação ou todos os mesmos defaults do registro v3.
Preserve model card/metrics/approval e metadata que a governança da equipe
exige.

### Experimentos e lineage

Recursos Core disponibilizam objetos para experimentos e lineage, como
contexts, actions, artifacts e associations. Mapeie quais dados/execuções
eram registrados na v2 e valide se o mesmo grafo, nomes, associação e
permissões são mantidos. Lineage não deve ser considerado automaticamente
idêntico só porque o nome do conceito persiste.

Para auditoria, registre pelo menos versão do código, imagem, dados de entrada,
parâmetros, nome/ARN de jobs, modelo registrado e execução de pipeline.

## 10. Feature Store

O Feature Store tem APIs v3 em `sagemaker.mlops.feature_store`, com recursos
base em Core. A documentação do checkout mostra `FeatureGroupManager`,
`FeatureDefinition`, ingestão de DataFrames, leitura de Online Store e
configurações Lake Formation/Iceberg.

| V2 | Direção v3 |
|---|---|
| `sagemaker.feature_store.feature_group.FeatureGroup` | `FeatureGroupManager` em `sagemaker.mlops.feature_store` e/ou `FeatureGroup` de `sagemaker.core.resources` |
| APIs de ingestão de FeatureGroup | funções/managers v3, por exemplo `ingest_dataframe` e `IngestionManagerPandas` |
| Leitura e operações da Feature Store | funções v3 e recursos Core; verificar o contrato da operação |

Exemplo de ingestão conceitual:

```python
from sagemaker.mlops.feature_store import ingest_dataframe

ingest_dataframe(
    feature_group_name="customer-features",
    data_frame=data,
    max_workers=4,
    max_processes=2,
)
```

Antes de migrar workloads:

- catalogue feature groups, schemas, record identifier, event time e stores
  online/offline;
- confirme consistência de nomes/tipos e opções de ingestão;
- preserve retenção, TTL, criptografia, catálogo e permissões Lake Formation;
- teste atraso/consistência entre online e offline e reconcilie contagens;
- avalie cuidadosamente mudanças one-way no tipo de armazenamento; o próprio
  guia v3 ressalta que certas migrações de storage não podem ser revertidas.

## 11. Monitoramento, Clarify e qualidade

O SDK v3 separa os recursos do serviço de monitoramento e análise em Core,
enquanto alguns workflows e integração ficam em MLOps. Clarify usa
`sagemaker.core.clarify` (por exemplo, `SageMakerClarifyProcessor`, `DataConfig`,
`BiasConfig`, `SHAPConfig`). Monitor jobs/schedules e quality-check steps
também têm shapes e steps próprios.

Migração não é só trocar import:

- replique baseline, janela, schedule, input/output e tipos de análise;
- compare métricas, constraints, relatórios e localização de artefatos;
- confira associações com endpoint/modelo e role que executa o monitor;
- confira KMS, tags e integração do output em ProcessingStep/ConditionStep;
- valide alertas/automação externos (EventBridge, CloudWatch, Lambda) que
  dependiam de nomes ou formatos antigos.

Monitoramento executado automaticamente e análise periódica pode gerar jobs
com custos recorrentes. Migre e teste schedules em ambiente controlado antes
de habilitá-los em produção.

## 12. SageMaker Core e chamadas de baixo nível

Para operações não cobertas pelas interfaces de alto nível, `sagemaker.core`
fornece recursos orientados a objetos, por exemplo `Model`, `Endpoint`,
`ProcessingJob`, `FeatureGroup`, `ModelPackage` e recursos de transformação.
Eles mapeiam operações para os serviços AWS e podem ajudar a substituir
chamadas diretas a boto3 ou APIs v2 quando o projeto precisa de controle
explícito.

Exemplo ilustrativo de recuperação de um recurso:

```python
from sagemaker.core.resources import ModelPackage

package = ModelPackage.get(model_package_name=registered_model_arn)
```

Use recursos Core quando for necessário gerir ciclo de vida ou consultar
detalhes que não estão expostos pelo builder de alto nível. Ao migrar código
boto3, confira:

- nomes de campos em snake_case versus nomes do request AWS;
- objetos tipados versus dicionários e comportamento de serialização;
- operações `get`, `create`, `refresh`, `update`, `delete` e paginação;
- exceções e consistência da sessão/região/credenciais;
- campos retornados como não atribuídos ou opcionais;
- necessidade de manter compatibilidade do JSON de requests/responses.

Uma abstração de recurso não garante que todos os detalhes internos do
serviço estejam automaticamente resolvidos. Use boto3 diretamente se a
operação necessária ainda não tiver wrapper apropriado ou se o contrato da
API AWS for o requisito explícito.

## 13. Novas capacidades e itens sem equivalência direta

V3 acrescenta ou expõe fluxos que não devem ser misturados com tarefas de
portabilidade:

- **Model customization/fine-tuning:** trainers especializados (SFT, DPO,
  RLAIF, RLVR e outros conforme o release) para técnicas de adaptação de
  foundation models. Planeje dados, formato, hardware, avaliação e custos
  como projeto novo.
- **`InferenceSpec` e in-process mode:** opções novas de integração de
  inferência customizada e execução; não são substituições obrigatórias do
  handler v2.
- **Integração MLflow e recursos de avaliação/AI Registry:** opções com
  maturidade e suporte dependentes do release e do serviço associado.
- **Steps adicionais de Pipeline:** por exemplo EMR Serverless, ForEach e
  Cleanup no checkout atual. Avalie compatibilidade da região, runtime e
  permissões separadamente.
- **Feature Store v3:** funcionalidades mais novas incluem novas operações e
  opções de configuração que podem exigir decisão de migração de dados ou
  storage.
- **JumpStart:** configurações e classes mudaram; use `JumpStartConfig` com
  builders/trainers v3, confirme disponibilidade do modelo na região, EULA,
  imagem, instância e permissões.

Recursos removidos ou sem suporte devem ser identificados cedo. O guia geral
do repositório lista algumas capacidades v2 sem equivalente atual (por
exemplo interfaces antigas específicas, certos estimadores/algoritmos e
configurações). Não interprete essa lista como um catálogo completo para toda
versão: faça um inventário por projeto e confirme com a documentação do
serviço e o release instalado.

## 14. Estratégia de migração para um projeto existente

### Fase A — inventário

Faça busca no código e liste todos os pontos de uso:

- `Estimator`, `Model`, `Predictor` e subclasses;
- deploy, endpoint, async/serverless/multi-model e serializers;
- Training, Processing e Batch Transform;
- `Pipeline`, steps, parâmetros, condições, cache e execution roles;
- tuning, registry, experiments, lineage e Model Cards;
- Feature Store, Clarify, Model Monitor, Quality/Baseline checks;
- `Session`, `get_execution_role`, `image_uris`, uploaders e clients boto3;
- configurações compartilhadas, defaults, imagens, tags, VPC, KMS e
  variáveis de ambiente.

Registre para cada fluxo: dono, ambiente, frequência, volume, custo,
criticidade, role, região, dependências, artefatos de entrada/saída e critério
de sucesso.

### Fase B — separar e fixar ambiente

1. Crie um ambiente/branch de migração; não atualize o ambiente de produção
   em place.
2. Fixe versões do SDK agregado e dependências diretas.
3. Garanta compatibilidade das imagens de treino/serving com as versões de
   Python/framework e arquitetura alvo.
4. Configure credenciais e região de teste, sem copiar secrets para o
   repositório.

### Fase C — migrar uma fatia vertical

Escolha um caso de uso não crítico e migre o fluxo completo, não apenas
imports: por exemplo, modelo registrado → endpoint de teste → chamada com
payload realista → limpeza. Em paralelo migre os outros componentes do
workload, preservando entradas, saídas e métricas. Os fluxos de treinamento e
processing têm guias próprios, vinculados no início deste relatório.

### Fase D — comparar comportamento

Execute v2 e v3 com os mesmos dados de teste, sementes e configurações
equivalentes. Compare:

- outputs, métricas e tolerâncias numéricas;
- formato do modelo e resposta da inferência;
- tamanho e localização de artefatos;
- parâmetros efetivamente enviados ao serviço;
- duração, logs, retries, estados de falha e custo;
- tags, metadados, lineage, aprovação e permissões.

Diferenças de framework/container, versão de algoritmo ou comportamento do
serviço podem causar alterações nos resultados mesmo quando o código Python
parece equivalente.

### Fase E — implantação e rollback

Implante v3 em recursos isolados com nomes próprios. Direcione tráfego de
teste, valide telemetria e monitore custos. Migre consumidores e schedulers
gradualmente. Mantenha o recurso v2 disponível até que a janela de rollback
seja encerrada e os artefatos v3 estejam confirmados.

### Fase F — limpeza

Depois da aprovação, retire endpoints, schedules, pipelines e recursos v2
que não são mais usados. Confirme que a retenção de modelos, logs, dados,
experimentos e lineage atende às políticas antes de remover qualquer dado.

## 15. Matriz de migração rápida

| Área v2 | Referência v2 comum | Direção v3 | Tipo |
|---|---|---|---|
| Deploy/modelo | `Model`, `PyTorchModel`, `TensorFlowModel` | `sagemaker.serve.ModelBuilder` + configuração/spec | Substituição |
| Predição | `Predictor` | endpoint/serving v3 e recursos Core | Substituição |
| Batch Transform | `sagemaker.transformer.Transformer` | `sagemaker.core.transformer.Transformer` | Reposicionamento |
| Pipeline | `sagemaker.workflow.pipeline.Pipeline` | `sagemaker.mlops.workflow.pipeline.Pipeline` | Reposicionamento |
| Steps | `sagemaker.workflow.steps.*` | `sagemaker.mlops.workflow.steps.*` | Reposicionamento |
| Pipeline session | `sagemaker.workflow.pipeline_context.PipelineSession` | `sagemaker.core.workflow.pipeline_context.PipelineSession` | Reposicionamento |
| Tuning | `sagemaker.tuner.HyperparameterTuner` | `sagemaker.train.tuner.HyperparameterTuner` + `sagemaker.core.parameter` | Substituição |
| Registry | `ModelPackage`, `ModelPackageGroup`, `register_model` | `ModelBuilder.register()` e recursos Core | Substituição |
| Feature Store | `sagemaker.feature_store.*` | `sagemaker.mlops.feature_store` + `sagemaker.core.resources` | Reposicionamento/substituição |
| Clarify | `sagemaker.clarify.*` | `sagemaker.core.clarify.*` | Reposicionamento |
| Monitoramento | `sagemaker.model_monitor.*` e schedules/check steps | shapes/recursos Core + steps MLOps | Reposicionamento; validar paridade |
| Experiments/lineage | `sagemaker.experiments.*`, `sagemaker.lineage.*` | `sagemaker.core.experiments` e `sagemaker.core.lineage` | Reposicionamento |
| Sessão | `sagemaker.Session()` | `sagemaker.core.helper.session_helper.Session()` | Reposicionamento |
| Imagens | `sagemaker.image_uris` | `sagemaker.core.image_uris` | Reposicionamento |
| Baixo nível | classes/helpers v2 ou boto3 | `sagemaker.core.resources.*` ou boto3 conforme necessidade | Escolha de abstração |
| Foundation model tuning | scripts/estimadores v2 diversos | trainers/model customization v3 | Nova capacidade ou redesenho |

## 16. Validação, riscos e checklist

### Checklist antes de iniciar

- [ ] Projeto executa em ambiente Python isolado com versões fixadas.
- [ ] Imports v2 restantes foram encontrados e classificados; cada exceção tem
  justificativa.
- [ ] Roles de cliente e execução foram verificadas, inclusive `iam:PassRole`.
- [ ] Imagens, versões, região, arquiteturas e tipos de instância foram
  confirmados.
- [ ] S3, ECR, KMS, VPC, isolamento e dependências de rede foram testados.
- [ ] Inputs/outputs e contratos de dados continuam equivalentes.
- [ ] Pipeline é definido e executado em teste com parâmetros/steps válidos.
- [ ] Endpoint de teste tem payloads e respostas comparados com v2.
- [ ] Tuning/monitoramento têm limite explícito de jobs, paralelismo e custo.
- [ ] Outputs, artefatos, tags, registro, lineage e logs atendem à governança.
- [ ] Há plano de rollback e processo de exclusão dos recursos temporários.

### Riscos a tratar

- **Migração de imports sem migração de comportamento:** o código importa, mas
  os defaults, imagem, resposta ou estrutura de artefato podem diferir.
- **APIs documentadas em versões distintas:** notebooks e docs do repositório
  podem usar caminhos diferentes. Use o manifest/versão efetivamente instalada
  como referência final.
- **Recursos persistentes e custos:** endpoints e schedules continuam ativos
  até serem excluídos/desabilitados.
- **Pipeline simbólico versus execução local:** valores de parâmetros e
  propriedades não estão necessariamente disponíveis como valores Python no
  momento em que a pipeline é construída.
- **Paridade incompleta:** alguns recursos podem não ter interface v3
  equivalente; registre workaround, uso direto de boto3 ou adiamento.
- **Dados e governança:** Feature Store, Model Registry, monitoramento e
  lineage podem exigir validação de metadados e de retenção além de testar o
  resultado numérico.

## 17. Referências do repositório

- [migration.md](./migration.md): guia amplo de migração V2 → V3; consulte-o
  junto com a API atual, pois alguns exemplos podem estar menos recentes que
  os módulos e documentação da versão atual.
- [CHANGELOG.md](./CHANGELOG.md): evolução por versão; o checkout analisado
  informa `3.27.0`.
- [docs/overview.rst](./docs/overview.rst): visão geral de arquitetura,
  pacotes, `ModelBuilder` e capacidades.
- [docs/inference/index.rst](./docs/inference/index.rst): serving, `InferenceSpec`,
  JumpStart, otimização e fluxo train-to-inference.
- [docs/ml_ops/index.rst](./docs/ml_ops/index.rst): Pipelines, Batch
  Transform, tuning, Model Registry, Clarify, MLflow e Feature Store.
- [docs/sagemaker_core/index.rst](./docs/sagemaker_core/index.rst): recursos
  Core e gestão de recursos AWS.
- [RELATORIO_TRAINING_SDK_V3.md](./RELATORIO_TRAINING_SDK_V3.md): detalhes de
  treinamento.
- [RELATORIO_PROCESSING_SDK_V3.md](./RELATORIO_PROCESSING_SDK_V3.md): detalhes
  de Processing.

**Recomendação final:** trate a migração como atualização de arquitetura e
valide cada fluxo ponta a ponta. Não remova a v2 nem recursos ativos até que
os resultados, permissões, operação e custos da v3 tenham sido aprovados.
